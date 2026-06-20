extension MLXModelPool {
    /// Registers or updates a model and its API-visible aliases.
    ///
    /// Re-registering the same model replaces its aliases. Registering a
    /// different model with an existing identifier fails.
    public func register(
        _ model: MLXLanguageModel,
        aliases: [String] = [],
        profiles: [MLXModelServingProfile] = []
    ) throws {
        let modelID = model.model.id
        if let existing = registrations[modelID], existing != model {
            throw MLXModelPoolError.duplicateModel(modelID)
        }
        try validateAliases(aliases, targetID: modelID, modelID: modelID)
        try validateServingProfiles(
            profiles,
            for: modelID,
            reservedAliases: Set(aliases.filter { $0 != modelID })
        )

        registrations[modelID] = model
        removeServingProfiles(for: modelID)
        aliasTargets = aliasTargets.filter { $0.value != modelID }
        for alias in aliases where alias != modelID {
            aliasTargets[alias] = modelID
        }
        for profile in profiles {
            let profileID = servingProfileID(modelID: modelID, profileName: profile.name)
            servingProfiles[profileID] = profile
            servingProfileTargets[profileID] = modelID
            for alias in profile.aliases where alias != profileID {
                aliasTargets[alias] = profileID
            }
        }
    }

    /// Removes a model registration and any idle resident sessions for it.
    public func unregister(id: String) async throws {
        let publicID = try resolvedPublicID(for: id)
        if servingProfiles[publicID] != nil {
            try await unload(id: publicID)
            servingProfiles.removeValue(forKey: publicID)
            servingProfileTargets.removeValue(forKey: publicID)
            aliasTargets = aliasTargets.filter { $0.value != publicID }
            return
        }

        let modelID = try canonicalModelID(for: publicID)
        try await unload(id: publicID)
        registrations.removeValue(forKey: modelID)
        let profileIDs = servingProfileTargets.compactMap { profileID, target -> String? in
            target == modelID ? profileID : nil
        }
        removeServingProfiles(for: modelID)
        aliasTargets = aliasTargets.filter { _, target in
            target != modelID && !profileIDs.contains(target)
        }
    }

    /// Returns the registered model for a canonical identifier or alias.
    public func model(id: String) throws -> MLXLanguageModel {
        let publicID = try resolvedPublicID(for: id)
        if let profile = servingProfiles[publicID],
            let modelID = servingProfileTargets[publicID],
            let model = registrations[modelID] {
            return profile.applying(to: model, publicID: publicID)
        }
        guard let model = registrations[publicID] else {
            throw MLXModelPoolError.unknownModel(id)
        }
        return model
    }
}

extension MLXModelPool {
    func validateAliases(
        _ aliases: [String],
        targetID: String,
        modelID: String
    ) throws {
        for alias in aliases where alias != targetID {
            if let existing = aliasTargets[alias],
                existing != targetID,
                !aliasTarget(existing, isOwnedBy: modelID) {
                throw MLXModelPoolError.aliasAlreadyRegistered(
                    alias: alias,
                    existingModelID: existing
                )
            }
            if registrations[alias] != nil, alias != targetID {
                throw MLXModelPoolError.aliasAlreadyRegistered(
                    alias: alias,
                    existingModelID: alias
                )
            }
            if servingProfiles[alias] != nil, alias != targetID {
                throw MLXModelPoolError.aliasAlreadyRegistered(
                    alias: alias,
                    existingModelID: alias
                )
            }
        }
    }

    func validateServingProfiles(
        _ profiles: [MLXModelServingProfile],
        for modelID: String,
        reservedAliases: Set<String>
    ) throws {
        var seenProfileIDs = Set<String>()
        var seenAliases = reservedAliases
        for profile in profiles {
            guard !profile.name.isEmpty else {
                throw MLXModelPoolError.invalidProfileName(modelID: modelID)
            }
            let profileID = servingProfileID(modelID: modelID, profileName: profile.name)
            guard seenProfileIDs.insert(profileID).inserted else {
                throw MLXModelPoolError.duplicateProfile(profileID)
            }
            try validateProfileID(profileID, modelID: modelID)
            try validateProfileAliases(profile, profileID: profileID, seenAliases: &seenAliases)
            try validateAliases(profile.aliases, targetID: profileID, modelID: modelID)
        }
    }

    func validateProfileID(
        _ profileID: String,
        modelID: String
    ) throws {
        if registrations[profileID] != nil {
            throw MLXModelPoolError.duplicateProfile(profileID)
        }
        if let existingTarget = aliasTargets[profileID],
            existingTarget != profileID,
            !aliasTarget(existingTarget, isOwnedBy: modelID) {
            throw MLXModelPoolError.aliasAlreadyRegistered(
                alias: profileID,
                existingModelID: existingTarget
            )
        }
    }

    func validateProfileAliases(
        _ profile: MLXModelServingProfile,
        profileID: String,
        seenAliases: inout Set<String>
    ) throws {
        for alias in profile.aliases where alias != profileID {
            guard seenAliases.insert(alias).inserted else {
                throw MLXModelPoolError.aliasAlreadyRegistered(
                    alias: alias,
                    existingModelID: profileID
                )
            }
        }
    }

    func removeServingProfiles(for modelID: String) {
        let profileIDs = servingProfileTargets.compactMap { profileID, target -> String? in
            target == modelID ? profileID : nil
        }
        for profileID in profileIDs {
            servingProfiles.removeValue(forKey: profileID)
            servingProfileTargets.removeValue(forKey: profileID)
        }
        aliasTargets = aliasTargets.filter { _, target in
            !profileIDs.contains(target)
        }
    }

    func servingProfileID(
        modelID: String,
        profileName: String
    ) -> String {
        "\(modelID):\(profileName)"
    }

    func aliasTarget(
        _ targetID: String,
        isOwnedBy modelID: String
    ) -> Bool {
        targetID == modelID || servingProfileTargets[targetID] == modelID
    }

    func canonicalModelID(for id: String) throws -> String {
        let publicID = try resolvedPublicID(for: id)
        if registrations[publicID] != nil {
            return publicID
        }
        if let target = servingProfileTargets[publicID] {
            return target
        }
        throw MLXModelPoolError.unknownModel(id)
    }

    func resolvedPublicID(for id: String) throws -> String {
        if registrations[id] != nil || servingProfiles[id] != nil {
            return id
        }
        if let target = aliasTargets[id] {
            return target
        }
        throw MLXModelPoolError.unknownModel(id)
    }
}
